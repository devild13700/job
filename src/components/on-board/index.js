'use client'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { TabsContent } from '@radix-ui/react-tabs'
import { useEffect, useState } from 'react'
import CommonForm from '../common-form'
import {
  candidateOnboardFormControls,
  initialCandidateFormData,
  initialRecruiterFormData,
  recruiterOnboardFormControls,
} from '@/utils'
import { useUser } from '@clerk/nextjs'
import { createProfileAction } from '@/actions'
import { createClient } from '@supabase/supabase-js'
import { v4 as uuidv4 } from 'uuid'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
const subbaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

const supabaseClient = createClient(supabaseUrl, subbaseKey)

function OnBoard() {
  const [currentTab, setCurrentTab] = useState('candidate')
  const [recruiterFormData, setRecruiterFormData] = useState(
    initialRecruiterFormData
  )
  const [candidateFormData, setCandidateFormData] = useState(
    initialCandidateFormData
  )
  const [file, setFile] = useState(null)

  const currentAuthUser = useUser()
  const { user } = currentAuthUser

  function handleFileChange(event) {
    event.preventDefault()
    setFile(event.target.files[0])
  }

  async function ExtractKeywords() {
    try {
      const userId = user?.id
      const resume = file

      if (!resume) {
        throw new Error('FileNotFound')
      }

      const formData = new FormData()
      formData.append('userId', userId)
      formData.append('resume', resume)
      const response = await fetch('/api/extract_keywords', {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()
      console.log(data)
    } catch (error) {
      console.error(error)
    }
  }

  async function handleUploadPdfToSupabase() {
    try {
      
      const filename = `${uuidv4()}-${Date.now()}.pdf`
      console.log(filename)
      const { data, error } = await supabaseClient.storage
        .from('workwise')
        .upload(`/public/${filename}`, file, {
          cacheControl: '3600',
          upsert: false,
        })
      console.log(data, error)
      if (data) {
        setCandidateFormData({
          ...candidateFormData,
          resume: data.path,
        })
      }
    } catch (error) {
      console.log(error)
    }
  }

  console.log(candidateFormData)

  useEffect(() => {
    if (file) {
      ExtractKeywords().then(()=>{
        handleUploadPdfToSupabase()
      })
      
    }
  }, [file])

  function handleTabChange(value) {
    setCurrentTab(value)
  }

  function handleRecuiterFormValid() {
    return (
      recruiterFormData &&
      recruiterFormData.name.trim() !== '' &&
      recruiterFormData.companyName.trim() !== '' &&
      recruiterFormData.companyRole.trim() !== ''
    )
  }

  function handleCandidateFormValid() {
    return Object.keys(candidateFormData).every(
      (key) => candidateFormData[key].trim() !== ''
    )
  }

  async function createProfile() {
    const data =
      currentTab === 'candidate'
        ? {
            candidateInfo: candidateFormData,
            role: 'candidate',
            isPremiumUser: false,
            userId: user?.id,
            email: user?.primaryEmailAddress?.emailAddress,
          }
        : {
            recruiterInfo: recruiterFormData,
            role: 'recruiter',
            isPremiumUser: false,
            userId: user?.id,
            email: user?.primaryEmailAddress?.emailAddress,
          }
        await  handleUploadPdfToSupabase()
    await createProfileAction(data, '/onboard')
  }

  console.log(candidateFormData)

  return (
    <div className='bg-white'>
      <Tabs value={currentTab} onValueChange={handleTabChange}>
        <div className='w-full'>
          <div className='flex items-baseline justify-between border-b pb-6 pt-24'>
            <h1 className='text-xl font-bold tracking-tight text-gray-900'>
              Welcome to Workwise
            </h1>
            <TabsList>
              <TabsTrigger value='candidate'>Candidate</TabsTrigger>
              <TabsTrigger value='recruiter'>Recruiter</TabsTrigger>
            </TabsList>
          </div>
        </div>
        <TabsContent value='candidate'>
          <CommonForm
            action={createProfile}
            formData={candidateFormData}
            setFormData={setCandidateFormData}
            formControls={candidateOnboardFormControls}
            buttonText={'Onboard as candidate'}
            handleFileChange={handleFileChange}
            isBtnDisabled={!handleCandidateFormValid()}
          />
        </TabsContent>
        <TabsContent value='recruiter'>
          <CommonForm
            formControls={recruiterOnboardFormControls}
            buttonText={'Onboard as recruiter'}
            formData={recruiterFormData}
            setFormData={setRecruiterFormData}
            isBtnDisabled={!handleRecuiterFormValid()}
            action={createProfile}
          />
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default OnBoard